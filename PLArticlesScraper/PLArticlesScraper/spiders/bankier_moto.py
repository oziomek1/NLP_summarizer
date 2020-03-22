# -*- coding: utf-8 -*-
import scrapy


class BankierMotoSpider(scrapy.Spider):
    name = 'bankier_moto'
    allowed_domains = ['bankier.pl']
    start_urls = ['http://bankier.pl/moto/wiadomosci/']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//section//div[contains(@class, 'row')]//div//div[contains(@class, 'row')]/div/a/@href"
    NEXT_PAGE_URL = "//footer/div/div/a/@href"

    TITLE_NODE = "//article/header/h1[contains(@class, 'header__title')]/text()"
    LEAD_NODE = "//article/div/div/p[contains(@class, 'lead')]//text()"
    TEXT_NODE = "//article/div/div/p/text()"

    def parse(self, response):
        # iterate through links to articles
        for quote in response.xpath(self.ARTICLES_URLS):
            yield response.follow(quote, callback=self.parse_article)
            
        #iterate through next page links
        for next_page in response.xpath(self.NEXT_PAGE_URL):
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract(),
            'text': response.xpath(self.TEXT_NODE).extract(),
        }
