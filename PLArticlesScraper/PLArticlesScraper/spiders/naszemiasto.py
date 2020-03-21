# -*- coding: utf-8 -*-
import scrapy


def get_category_start_urls():
	return [
    	'http://naszemiasto.pl/',
    	'https://wroclaw.naszemiasto.pl/',
    	'https://warszawa.naszemiasto.pl/',
    	'https://krakow.naszemiasto.pl/',
    	'https://poznan.naszemiasto.pl/',
    	'https://lodz.naszemiasto.pl/',
    	'https://katowice.naszemiasto.pl/',
    ]


class NaszemiastoSpider(scrapy.Spider):
    name = 'naszemiasto'
    allowed_domains = ['naszemiasto.pl']
    start_urls = get_category_start_urls()

    custom_settings = {
        'DEPTH_LIMIT': 0
    }
    ARTICLES_URLS = "//article[contains(@class, 'kafel')]/a/@href | //article[contains(@class, 'kafla')]/a/@href | //article[contains(@class, 'kafla')]/div/a/@href | //ul[contains(@class, 'listaMaterialow')]/li//a/@href"
    
    ARTICLE_SIDE_ARTICLES_URLS = "//div[contains(@class, 'componentsRecommendationsMixedListing__elementWithTile')]/div/a[contains(@class, 'atomsArticleTile__info')]"
    ARTICLE_BOTTOM_ARTICLES_URLS = "//div[contains(@class, 'componentsRecommendationsSimpleListing__element')]/div/a[contains(@class, 'atomsArticleTile__info')]"

    TITLE_NODE = "//header/h1/text()"
    LEAD_NODE = "//div[contains(@class, 'componentsArticleLead')]/text()"
    TEXT_NODE = "//article[contains(@class, 'componentsArticleContent')]/div[contains(@class, 'md')]/p//text() | //article[contains(@class, 'componentsArticleContent')]/div[contains(@class, 'md')]/h2"

    def parse(self, response):
        # iterate through links to articles
        for quote in [
        	*response.xpath(self.ARTICLES_URLS),
        	*response.xpath(self.ARTICLE_SIDE_ARTICLES_URLS),
        	*response.xpath(self.ARTICLE_BOTTOM_ARTICLES_URLS),
        ]:
            yield response.follow(quote, callback=self.parse_article)
            
        #iterate through next page links
        for next_page in [
        	*response.xpath(self.ARTICLE_SIDE_ARTICLES_URLS),
        	*response.xpath(self.ARTICLE_BOTTOM_ARTICLES_URLS),
        ]:
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract(),
            'text': response.xpath(self.TEXT_NODE).extract(),
        }
