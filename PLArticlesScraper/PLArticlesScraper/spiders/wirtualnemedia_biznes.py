# -*- coding: utf-8 -*-
import scrapy


class WirtualnemediaBiznesSpider(scrapy.Spider):
    name = 'wirtualnemedia_biznes'
    allowed_domains = ['wirtualnemedia.pl']
    start_urls = ['https://www.wirtualnemedia.pl/archiwum/biznes']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//div[contains(@class, 'subsite-row')]/div/a/@href"
    NEXT_PAGE_URL = "//li[contains(@class, 'next-link')]/a/@href"

    TITLE_NODE = "//div[contains(@class, 'page-content')]//h1/text()"
    LEAD_NODE = "//div[contains(@class, 'page-content')]//div[contains(@class, 'top-column')]//p/text()"
    TEXT_NODE = "//div[contains(@class, 'page-content')]//article/p//text()"

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
